"""
WebSocket connection manager
"""
import logging
import json
from typing import Set, Dict, Any
from fastapi import WebSocket
from app.config import settings

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.kafka_consumer = None
    
    async def initialize(self):
        """Initialize WebSocket manager and Kafka consumer"""
        try:
            # Initialize Kafka consumer for real-time updates
            from app.websocket.kafka_consumer import KafkaWebSocketConsumer
            self.kafka_consumer = KafkaWebSocketConsumer(self)
            await self.kafka_consumer.start()
            logger.info("Kafka consumer pour WebSocket démarré")
        except Exception as e:
            logger.warning(f"Impossible de démarrer le consumer Kafka: {e}. WebSocket fonctionnera sans mises à jour temps-réel.")
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connecté. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket déconnecté. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Erreur lors de l'envoi d'un message WebSocket: {e}")
            await self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSocket clients"""
        if not self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Erreur lors du broadcast WebSocket: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.active_connections.discard(connection)
    
    async def broadcast_feature_update(self, asset_id: str, features: Dict[str, Any]):
        """Broadcast feature update"""
        from datetime import datetime
        await self.broadcast({
            "type": "feature_update",
            "asset_id": asset_id,
            "data": features,
            "timestamp": datetime.now().isoformat()
        })
    
    async def broadcast_anomaly_detected(self, anomaly: Dict[str, Any]):
        """Broadcast anomaly detection"""
        from datetime import datetime
        await self.broadcast({
            "type": "anomaly_detected",
            "data": anomaly,
            "timestamp": datetime.now().isoformat()
        })
    
    async def broadcast_rul_prediction(self, prediction: Dict[str, Any]):
        """Broadcast RUL prediction"""
        from datetime import datetime
        await self.broadcast({
            "type": "rul_prediction",
            "data": prediction,
            "timestamp": datetime.now().isoformat()
        })
    
    async def cleanup(self):
        """Cleanup WebSocket manager"""
        # Close all connections
        for connection in list(self.active_connections):
            try:
                await connection.close()
            except:
                pass
        self.active_connections.clear()
        
        # Stop Kafka consumer
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        
        logger.info("WebSocket manager nettoyé")

