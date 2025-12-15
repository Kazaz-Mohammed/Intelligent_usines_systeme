"""
Application FastAPI principale pour le service Dashboard Usine
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.config import settings
from app.websocket.websocket_manager import WebSocketManager

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation du WebSocket manager
websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    logger.info(f"Démarrage du service {settings.service_name}")
    logger.info(f"Port: {settings.service_port}")
    logger.info(f"Database: {settings.database_host}:{settings.database_port}/{settings.database_name}")
    logger.info(f"WebSocket path: {settings.websocket_path}")
    
    # Initialiser le WebSocket manager
    await websocket_manager.initialize()
    logger.info("WebSocket manager initialisé")
    
    yield
    
    # Shutdown
    logger.info(f"Arrêt du service {settings.service_name}")
    await websocket_manager.cleanup()
    logger.info("WebSocket manager arrêté")


app = FastAPI(
    title="Dashboard Usine Service",
    description="Service de visualisation temps-réel pour la maintenance prédictive",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from app.api import assets, anomalies, rul, interventions, kpis, gis, export, grafana

app.include_router(assets.router, prefix="/api/v1/assets", tags=["Assets"])
app.include_router(anomalies.router, prefix="/api/v1/anomalies", tags=["Anomalies"])
app.include_router(rul.router, prefix="/api/v1/rul", tags=["RUL"])
app.include_router(interventions.router, prefix="/api/v1/interventions", tags=["Interventions"])
app.include_router(kpis.router, prefix="/api/v1/kpis", tags=["KPIs"])
app.include_router(gis.router, prefix="/api/v1/gis", tags=["GIS"])
app.include_router(export.router, prefix="/api/v1/export", tags=["Export"])
app.include_router(grafana.router, prefix="/api/v1/grafana", tags=["Grafana"])

# Health check
@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": "0.1.0"
    }

# Simple test WebSocket endpoint
from fastapi import WebSocket as WSTest
@app.websocket("/ws/test")
async def websocket_test(websocket: WSTest):
    """Simple test WebSocket endpoint"""
    print(">>> TEST WebSocket endpoint called!", flush=True)
    logger.info(">>> TEST WebSocket endpoint called!")
    await websocket.accept()
    await websocket.send_json({"message": "Hello from test WebSocket!"})
    await websocket.close()

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Dashboard Usine Service is running",
        "version": "0.1.0",
        "docs": "/docs"
    }

# WebSocket endpoint - MUST be registered before static file mounts
from app.websocket.websocket_handler import websocket_endpoint, set_websocket_manager
from fastapi import WebSocket

# Set the websocket manager in the handler
set_websocket_manager(websocket_manager)

# Register WebSocket route using decorator syntax (most reliable method)
# Using hardcoded path to ensure it works
@app.websocket("/ws/dashboard")
async def websocket_route(websocket: WebSocket):
    """WebSocket route handler"""
    logger.info(f"=== WebSocket route /ws/dashboard called! Client: {websocket.client} ===")
    print(f"=== WebSocket route /ws/dashboard called! Client: {websocket.client} ===", flush=True)
    await websocket_endpoint(websocket)

logger.info(f"WebSocket route registered at: /ws/dashboard (hardcoded)")

# Also list all routes for debugging
for route in app.routes:
    logger.info(f"Route: {route.path if hasattr(route, 'path') else route}")

# Serve static files in production (frontend build)
# Note: This must be AFTER WebSocket routes to avoid conflicts
# We don't mount static files at root to avoid interfering with API and WebSocket routes
try:
    app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")
    # Don't mount root to avoid conflicts with WebSocket and API routes
    # Frontend should be served by Next.js dev server or a separate web server in production
except Exception as e:
    logger.warning(f"Frontend static files not found: {e}. Running in API-only mode.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=False,
        log_level=settings.log_level.lower()
    )

