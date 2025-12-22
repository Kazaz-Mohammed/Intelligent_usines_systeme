"""
API endpoints for data export (PDF, CSV)
"""
import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
import io
import csv
from app.config import settings
from app.services.anomaly_service import AnomalyService
from app.services.rul_service import RULService
from app.services.orchestrator_service import OrchestratorService
from app.services.extraction_service import ExtractionService
from app.services.kpi_service import KPIService
from app.database.postgresql import PostgreSQLService, get_postgresql_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
anomaly_service = AnomalyService()
rul_service = RULService()
orchestrator_service = OrchestratorService()
extraction_service = ExtractionService()


@router.get("/csv")
async def export_csv(
    asset_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    data_type: str = Query("features", regex="^(features|anomalies|rul|interventions|all)$")
):
    """Export data to CSV"""
    try:
        # Parse date strings to datetime objects
        start_dt = None
        end_dt = None
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                try:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                except ValueError:
                    logger.warning(f"Invalid start_date format: {start_date}")
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                try:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    logger.warning(f"Invalid end_date format: {end_date}")
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Handle "all" data type by exporting all types
        data_types = ["features", "anomalies", "rul", "interventions"] if data_type == "all" else [data_type]
        
        for dt in data_types:
            if dt == "features":
                writer.writerow(["=== FEATURES ==="])
                writer.writerow(["asset_id", "sensor_id", "timestamp", "feature_name", "value"])
                
                # Fetch features - need to get all assets if no asset_id specified
                if asset_id:
                    try:
                        result = await extraction_service.get_features(
                            asset_id=asset_id,
                            start_time=start_dt,
                            end_time=end_dt,
                            limit=settings.export_max_records
                        )
                        features = result.get("features", [])
                        logger.info(f"Found {len(features)} features for asset {asset_id}")
                        
                        if not features:
                            writer.writerow([f"No features found for asset {asset_id} in the specified date range"])
                        else:
                            for feature in features:
                                feature_name = feature.get("name", "")
                                value = feature.get("value", "")
                                metadata = feature.get("metadata", {})
                                timestamp = metadata.get("timestamp", "")
                                if isinstance(timestamp, datetime):
                                    timestamp = timestamp.isoformat()
                                elif isinstance(timestamp, str):
                                    pass  # Already a string
                                sensor_id = metadata.get("sensor_id", "")
                                writer.writerow([asset_id, sensor_id, str(timestamp), feature_name, str(value)])
                    except Exception as e:
                        logger.error(f"Error fetching features for asset {asset_id}: {e}", exc_info=True)
                        writer.writerow([f"Error fetching features for asset {asset_id}: {str(e)}"])
                else:
                    # Query database directly for all assets
                    try:
                        db = await get_postgresql_service()
                        query = "SELECT asset_id, sensor_id, timestamp, feature_name, feature_value FROM extracted_features WHERE 1=1"
                        params = []
                        param_num = 1
                        
                        if start_dt:
                            query += f" AND timestamp >= ${param_num}"
                            params.append(start_dt)
                            param_num += 1
                        if end_dt:
                            query += f" AND timestamp <= ${param_num}"
                            params.append(end_dt)
                            param_num += 1
                        
                        query += f" ORDER BY timestamp DESC LIMIT ${param_num}"
                        params.append(settings.export_max_records)
                        
                        rows = await db.fetch(query, *params)
                        logger.info(f"Found {len(rows)} features from database")
                        
                        if not rows:
                            writer.writerow(["No features found in the specified date range"])
                        else:
                            for row in rows:
                                writer.writerow([
                                    row.get("asset_id", ""),
                                    row.get("sensor_id", ""),
                                    str(row.get("timestamp", "")),
                                    row.get("feature_name", ""),
                                    str(row.get("feature_value", ""))
                                ])
                    except Exception as e:
                        logger.error(f"Error fetching features from database: {e}", exc_info=True)
                        writer.writerow([f"Error fetching features: {str(e)}"])
                    
            elif dt == "anomalies":
                writer.writerow([])
                writer.writerow(["=== ANOMALIES ==="])
                writer.writerow(["asset_id", "sensor_id", "timestamp", "severity", "is_anomaly", "final_score", "criticality"])
                
                result = await anomaly_service.get_anomalies(
                    asset_id=asset_id,
                    start_date=start_dt,
                    end_date=end_dt,
                    limit=settings.export_max_records
                )
                anomalies = result.get("anomalies", result.get("items", []))
                for anomaly in anomalies:
                    writer.writerow([
                        anomaly.get("asset_id", ""),
                        anomaly.get("sensor_id", ""),
                        anomaly.get("timestamp", ""),
                        anomaly.get("severity", anomaly.get("criticality", "")),
                        anomaly.get("is_anomaly", False),
                        anomaly.get("final_score", ""),
                        anomaly.get("criticality", "")
                    ])
                    
            elif dt == "rul":
                writer.writerow([])
                writer.writerow(["=== RUL PREDICTIONS ==="])
                writer.writerow(["asset_id", "sensor_id", "timestamp", "rul_prediction", "confidence_interval_lower", "confidence_interval_upper", "model_used"])
                
                result = await rul_service.get_rul_predictions(
                    asset_id=asset_id,
                    start_date=start_dt,
                    end_date=end_dt,
                    limit=settings.export_max_records
                )
                predictions = result.get("predictions", [])
                for pred in predictions:
                    writer.writerow([
                        pred.get("asset_id", ""),
                        pred.get("sensor_id", ""),
                        pred.get("timestamp", ""),
                        pred.get("rul_prediction", ""),
                        pred.get("confidence_interval_lower", ""),
                        pred.get("confidence_interval_upper", ""),
                        pred.get("model_used", "")
                    ])
                    
            elif dt == "interventions":
                writer.writerow([])
                writer.writerow(["=== INTERVENTIONS ==="])
                writer.writerow(["asset_id", "title", "status", "priority", "type", "scheduled_start", "scheduled_end", "actual_start", "actual_end"])
                
                result = await orchestrator_service.get_interventions(
                    asset_id=asset_id,
                    limit=settings.export_max_records
                )
                interventions = result.get("interventions", [])
                for interv in interventions:
                    writer.writerow([
                        interv.get("asset_id", ""),
                        interv.get("title", ""),
                        interv.get("status", ""),
                        interv.get("priority", ""),
                        interv.get("type", interv.get("intervention_type", "")),
                        interv.get("scheduled_start", ""),
                        interv.get("scheduled_end", ""),
                        interv.get("actual_start", ""),
                        interv.get("actual_end", "")
                    ])
        
        output.seek(0)
        filename = f"export_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            iter([output.getvalue().encode('utf-8')]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Erreur lors de l'export CSV: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pdf")
async def export_pdf(
    asset_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    report_type: str = Query("summary", regex="^(summary|detailed|kpi)$")
):
    """Export report to PDF"""
    try:
        # Check if reportlab is available
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
        except ImportError:
            logger.error("reportlab is not installed. Please install it: pip install reportlab")
            raise HTTPException(status_code=500, detail="PDF generation requires reportlab library. Please install it.")
        
        # Parse date strings to datetime objects
        start_dt = None
        end_dt = None
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                try:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                except ValueError:
                    logger.warning(f"Invalid start_date format: {start_date}")
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                try:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    logger.warning(f"Invalid end_date format: {end_date}")
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        y_position = 750
        
        # Title
        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, y_position, f"Maintenance Report - {report_type.title()}")
        y_position -= 30
        
        # Report metadata
        p.setFont("Helvetica", 12)
        if asset_id:
            p.drawString(100, y_position, f"Asset ID: {asset_id}")
            y_position -= 20
        if start_dt:
            p.drawString(100, y_position, f"Start Date: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            y_position -= 20
        if end_dt:
            p.drawString(100, y_position, f"End Date: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            y_position -= 20
        
        y_position -= 20
        
        # Helper function to check if we need a new page
        def check_new_page(current_y, margin=100):
            if current_y < margin:
                p.showPage()
                return 750
            return current_y
        
        # Fetch and add data based on report type
        if report_type == "summary":
            # Summary statistics with detailed tables
            p.setFont("Helvetica-Bold", 14)
            p.drawString(100, y_position, "Summary Statistics")
            y_position -= 25
            
            p.setFont("Helvetica", 10)
            # Get anomaly count and data
            anomalies_result = await anomaly_service.get_anomalies(
                asset_id=asset_id,
                start_date=start_dt,
                end_date=end_dt,
                limit=50  # Get more data for detailed view
            )
            anomaly_count = anomalies_result.get("total", 0)
            anomalies = anomalies_result.get("anomalies", anomalies_result.get("items", []))
            p.drawString(100, y_position, f"Total Anomalies: {anomaly_count}")
            y_position -= 20
            
            # Get RUL predictions count and data
            rul_result = await rul_service.get_rul_predictions(
                asset_id=asset_id,
                start_date=start_dt,
                end_date=end_dt,
                limit=50  # Get more data for detailed view
            )
            rul_count = rul_result.get("total", 0)
            rul_predictions = rul_result.get("predictions", [])
            p.drawString(100, y_position, f"Total RUL Predictions: {rul_count}")
            y_position -= 20
            
            # Get interventions count and data
            interventions_result = await orchestrator_service.get_interventions(
                asset_id=asset_id,
                limit=50  # Get more data for detailed view
            )
            intervention_count = interventions_result.get("total", 0)
            interventions = interventions_result.get("interventions", [])
            p.drawString(100, y_position, f"Total Interventions: {intervention_count}")
            y_position -= 30
            
            # Anomalies Table
            if anomalies:
                y_position = check_new_page(y_position)
                p.setFont("Helvetica-Bold", 12)
                p.drawString(100, y_position, "Recent Anomalies")
                y_position -= 20
                
                # Table header
                p.setFont("Helvetica-Bold", 9)
                p.drawString(100, y_position, "Timestamp")
                p.drawString(200, y_position, "Sensor")
                p.drawString(280, y_position, "Criticality")
                p.drawString(360, y_position, "Score")
                y_position -= 15
                
                # Table rows
                p.setFont("Helvetica", 8)
                for anomaly in anomalies[:20]:  # Limit to 20 rows
                    y_position = check_new_page(y_position, 50)
                    timestamp_str = str(anomaly.get('timestamp', ''))[:16]  # Truncate timestamp
                    sensor_id = str(anomaly.get('sensor_id', 'N/A'))[:15]
                    criticality = str(anomaly.get('criticality', anomaly.get('severity', 'N/A')))[:10]
                    score = f"{anomaly.get('final_score', 0):.3f}"
                    
                    p.drawString(100, y_position, timestamp_str)
                    p.drawString(200, y_position, sensor_id)
                    p.drawString(280, y_position, criticality)
                    p.drawString(360, y_position, score)
                    y_position -= 12
                y_position -= 10
            
            # RUL Predictions Table
            if rul_predictions:
                y_position = check_new_page(y_position)
                p.setFont("Helvetica-Bold", 12)
                p.drawString(100, y_position, "Recent RUL Predictions")
                y_position -= 20
                
                # Table header
                p.setFont("Helvetica-Bold", 9)
                p.drawString(100, y_position, "Timestamp")
                p.drawString(200, y_position, "Sensor")
                p.drawString(280, y_position, "RUL (h)")
                p.drawString(340, y_position, "CI Lower")
                p.drawString(400, y_position, "CI Upper")
                p.drawString(470, y_position, "Model")
                y_position -= 15
                
                # Table rows
                p.setFont("Helvetica", 8)
                for pred in rul_predictions[:20]:  # Limit to 20 rows
                    y_position = check_new_page(y_position, 50)
                    timestamp_str = str(pred.get('timestamp', ''))[:16]
                    sensor_id = str(pred.get('sensor_id', 'N/A'))[:15]
                    rul = f"{pred.get('rul_prediction', 0):.1f}"
                    ci_lower = f"{pred.get('confidence_interval_lower', 0):.1f}" if pred.get('confidence_interval_lower') else "N/A"
                    ci_upper = f"{pred.get('confidence_interval_upper', 0):.1f}" if pred.get('confidence_interval_upper') else "N/A"
                    model = str(pred.get('model_used', 'N/A'))[:10]
                    
                    p.drawString(100, y_position, timestamp_str)
                    p.drawString(200, y_position, sensor_id)
                    p.drawString(280, y_position, rul)
                    p.drawString(340, y_position, ci_lower)
                    p.drawString(400, y_position, ci_upper)
                    p.drawString(470, y_position, model)
                    y_position -= 12
                y_position -= 10
            
            # Interventions Table
            if interventions:
                y_position = check_new_page(y_position)
                p.setFont("Helvetica-Bold", 12)
                p.drawString(100, y_position, "Recent Interventions")
                y_position -= 20
                
                # Table header
                p.setFont("Helvetica-Bold", 9)
                p.drawString(100, y_position, "Title")
                p.drawString(250, y_position, "Status")
                p.drawString(320, y_position, "Priority")
                p.drawString(390, y_position, "Scheduled Start")
                y_position -= 15
                
                # Table rows
                p.setFont("Helvetica", 8)
                for interv in interventions[:20]:  # Limit to 20 rows
                    y_position = check_new_page(y_position, 50)
                    title = str(interv.get('title', 'N/A'))[:30]
                    status = str(interv.get('status', 'N/A'))[:10]
                    priority = str(interv.get('priority', 'N/A'))[:10]
                    scheduled_start = str(interv.get('scheduled_start', 'N/A'))[:16] if interv.get('scheduled_start') else "N/A"
                    
                    p.drawString(100, y_position, title)
                    p.drawString(250, y_position, status)
                    p.drawString(320, y_position, priority)
                    p.drawString(390, y_position, scheduled_start)
                    y_position -= 12
            
        elif report_type == "detailed":
            # Detailed data tables (same as summary but with more rows)
            # Anomalies
            y_position = check_new_page(y_position)
            p.setFont("Helvetica-Bold", 12)
            p.drawString(100, y_position, "Anomalies")
            y_position -= 20
            
            p.setFont("Helvetica", 9)
            anomalies_result = await anomaly_service.get_anomalies(
                asset_id=asset_id,
                start_date=start_dt,
                end_date=end_dt,
                limit=100
            )
            anomalies = anomalies_result.get("anomalies", anomalies_result.get("items", []))
            
            # Table header
            p.setFont("Helvetica-Bold", 9)
            p.drawString(100, y_position, "Timestamp")
            p.drawString(200, y_position, "Sensor")
            p.drawString(280, y_position, "Criticality")
            p.drawString(360, y_position, "Score")
            p.drawString(420, y_position, "Anomaly")
            y_position -= 15
            
            # Table rows
            p.setFont("Helvetica", 8)
            for anomaly in anomalies[:50]:  # More rows for detailed
                y_position = check_new_page(y_position, 50)
                timestamp_str = str(anomaly.get('timestamp', ''))[:16]
                sensor_id = str(anomaly.get('sensor_id', 'N/A'))[:15]
                criticality = str(anomaly.get('criticality', anomaly.get('severity', 'N/A')))[:10]
                score = f"{anomaly.get('final_score', 0):.3f}"
                is_anomaly = "Yes" if anomaly.get('is_anomaly', False) else "No"
                
                p.drawString(100, y_position, timestamp_str)
                p.drawString(200, y_position, sensor_id)
                p.drawString(280, y_position, criticality)
                p.drawString(360, y_position, score)
                p.drawString(420, y_position, is_anomaly)
                y_position -= 12
        
        elif report_type == "kpi":
            # KPI Report
            db = await get_postgresql_service()
            kpi_service = KPIService(db)
            
            # Calculate days from date range
            days = 30
            if start_dt and end_dt:
                days = (end_dt - start_dt).days
            elif start_dt:
                days = (datetime.now() - start_dt).days
            
            kpi_summary = await kpi_service.get_kpi_summary(asset_id, days)
            
            y_position = check_new_page(y_position)
            p.setFont("Helvetica-Bold", 14)
            p.drawString(100, y_position, "KPI Metrics")
            y_position -= 30
            
            p.setFont("Helvetica-Bold", 11)
            p.drawString(100, y_position, "Key Performance Indicators")
            y_position -= 25
            
            p.setFont("Helvetica", 10)
            if kpi_summary.mtbf is not None:
                p.drawString(100, y_position, f"MTBF (Mean Time Between Failures): {kpi_summary.mtbf:.2f} hours")
            else:
                p.drawString(100, y_position, "MTBF (Mean Time Between Failures): N/A (no failures recorded)")
            y_position -= 20
            
            if kpi_summary.mttr is not None:
                p.drawString(100, y_position, f"MTTR (Mean Time To Repair): {kpi_summary.mttr:.2f} hours")
            else:
                p.drawString(100, y_position, "MTTR (Mean Time To Repair): N/A")
            y_position -= 20
            
            if kpi_summary.oee is not None:
                p.drawString(100, y_position, f"OEE (Overall Equipment Effectiveness): {kpi_summary.oee:.2f}%")
            else:
                p.drawString(100, y_position, "OEE (Overall Equipment Effectiveness): N/A")
            y_position -= 20
            
            if kpi_summary.availability is not None:
                p.drawString(100, y_position, f"Availability: {kpi_summary.availability:.2f}%")
            else:
                p.drawString(100, y_position, "Availability: N/A")
            y_position -= 20
            
            if kpi_summary.reliability is not None:
                p.drawString(100, y_position, f"Reliability: {kpi_summary.reliability:.2f}%")
            else:
                p.drawString(100, y_position, "Reliability: N/A")
            y_position -= 30
            
            # Add interpretation
            p.setFont("Helvetica-Bold", 11)
            p.drawString(100, y_position, "Interpretation")
            y_position -= 20
            
            p.setFont("Helvetica", 9)
            if kpi_summary.oee is not None:
                if kpi_summary.oee >= 85:
                    interpretation = "Excellent performance"
                elif kpi_summary.oee >= 70:
                    interpretation = "Good performance"
                elif kpi_summary.oee >= 50:
                    interpretation = "Average performance - improvement needed"
                else:
                    interpretation = "Poor performance - urgent action required"
                p.drawString(100, y_position, f"OEE Status: {interpretation}")
                y_position -= 15
        
        # Footer
        p.setFont("Helvetica", 8)
        p.drawString(100, 50, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        p.showPage()
        p.save()
        
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        if not pdf_data:
            logger.error("PDF buffer is empty after generation")
            raise HTTPException(status_code=500, detail="Failed to generate PDF content")
        
        filename = f"report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return StreamingResponse(
            iter([pdf_data]),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except ImportError as e:
        logger.error(f"reportlab import error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation library not available: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur lors de l'export PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

