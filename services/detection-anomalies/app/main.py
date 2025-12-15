"""
Service Detection Anomalies - Point d'entrée FastAPI
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import anomalies
import logging

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="Detection Anomalies Service",
    description="Service de détection d'anomalies en temps-réel pour la maintenance prédictive",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routers
app.include_router(anomalies.router)


@app.get("/")
async def root():
    """Point d'entrée du service"""
    return {
        "message": "Detection Anomalies Service is running",
        "service": settings.service_name,
        "version": "0.1.0",
        "endpoints": {
            "detect": "/api/v1/anomalies/detect",
            "detect_batch": "/api/v1/anomalies/detect/batch",
            "train": "/api/v1/anomalies/train",
            "status": "/api/v1/anomalies/status",
            "anomalies": "/api/v1/anomalies/",
            "metrics": "/api/v1/anomalies/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": "0.1.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=True
    )

