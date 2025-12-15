"""
Application FastAPI principale
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.preprocessing import router as preprocessing_router

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    logger.info(f"Démarrage du service {settings.service_name}")
    logger.info(f"Configuration Kafka: {settings.kafka_bootstrap_servers}")
    logger.info(f"Topic input: {settings.kafka_topic_input}")
    logger.info(f"Topic output: {settings.kafka_topic_output}")
    
    yield
    
    # Shutdown
    logger.info(f"Arrêt du service {settings.service_name}")


# Créer l'application FastAPI
app = FastAPI(
    title="Service Prétraitement",
    description="Service de nettoyage et normalisation des données capteurs",
    version="0.3.0",
    lifespan=lifespan
)

# Inclure les routers
app.include_router(preprocessing_router, prefix="/api/v1/preprocessing", tags=["preprocessing"])


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "service": settings.service_name,
        "version": "0.3.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "UP",
        "service": settings.service_name
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=True
    )
