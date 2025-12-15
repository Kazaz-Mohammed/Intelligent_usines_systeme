"""
Endpoints REST pour le service Prétraitement
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.config import settings

router = APIRouter()


@router.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "UP",
        "service": settings.service_name
    }


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Status endpoint avec informations détaillées"""
    return {
        "service": settings.service_name,
        "status": "running",
        "version": "0.3.0",
        "kafka": {
            "bootstrap_servers": settings.kafka_bootstrap_servers,
            "topic_input": settings.kafka_topic_input,
            "topic_output": settings.kafka_topic_output,
            "consumer_group": settings.kafka_consumer_group
        },
        "preprocessing": {
            "window_size": settings.window_size,
            "window_overlap": settings.window_overlap,
            "outlier_threshold": settings.outlier_threshold,
            "enable_denoising": settings.enable_denoising,
            "enable_frequency_analysis": settings.enable_frequency_analysis
        }
    }


@router.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Metrics endpoint (à implémenter avec Prometheus)"""
    return {
        "messages_processed": 0,  # TODO: Implémenter compteurs
        "messages_published": 0,
        "errors": 0,
        "processing_time_avg_ms": 0.0
    }

