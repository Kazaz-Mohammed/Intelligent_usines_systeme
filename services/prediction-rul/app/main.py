"""
Point d'entrée FastAPI pour le service Prediction RUL
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api import rul

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    logger.info(f"Démarrage du service {settings.service_name}")
    logger.info(f"Port: {settings.service_port}")
    logger.info(f"Kafka: {settings.kafka_bootstrap_servers}")
    logger.info(f"MLflow: {settings.mlflow_tracking_uri}")
    
    yield
    
    # Shutdown
    logger.info(f"Arrêt du service {settings.service_name}")


app = FastAPI(
    title="Prediction RUL Service",
    description="Service de prédiction de la Remaining Useful Life (RUL) pour la maintenance prédictive",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(rul.router, prefix="/api/v1/rul", tags=["RUL Prediction"])


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "service": settings.service_name,
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": "0.1.0"
    }

