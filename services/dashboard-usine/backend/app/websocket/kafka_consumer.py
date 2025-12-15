"""
Kafka consumer for WebSocket real-time updates
"""
import logging
import json
import asyncio
import threading
from kafka import KafkaConsumer
from typing import Dict, Any
from app.config import settings
from app.websocket.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class KafkaWebSocketConsumer:
    """Consumes Kafka messages and broadcasts via WebSocket"""
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self.consumer = None
        self.running = False
        self.thread = None
        self.main_loop = None
    
    async def start(self):
        """Start Kafka consumer"""
        try:
            topics = settings.kafka_topics.split(',')
            topics = [topic.strip() for topic in topics]
            
            self.consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=settings.kafka_bootstrap_servers.split(','),
                group_id=f"{settings.kafka_consumer_group}-websocket",
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            # Store reference to main event loop
            self.main_loop = asyncio.get_event_loop()
            
            self.running = True
            # Run the blocking poll in a thread
            self.thread = threading.Thread(target=self._consume_loop, daemon=True)
            self.thread.start()
            logger.info(f"Consumer Kafka démarré pour les topics: {topics}")
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du consumer Kafka: {e}", exc_info=True)
            raise
    
    def _consume_loop(self):
        """Main consumption loop (runs in separate thread)"""
        while self.running:
            try:
                # Poll for messages (blocking call, but in separate thread)
                message_pack = self.consumer.poll(timeout_ms=1000)
                
                if message_pack and self.main_loop:
                    # Schedule async processing on main event loop
                    for topic_partition, messages in message_pack.items():
                        for message in messages:
                            future = asyncio.run_coroutine_threadsafe(
                                self._process_message(topic_partition.topic, message.value),
                                self.main_loop
                            )
                            # Don't wait for completion to avoid blocking
                            # Errors will be logged in _process_message
            except Exception as e:
                logger.error(f"Erreur dans la boucle de consommation Kafka: {e}", exc_info=True)
                import time
                time.sleep(1)
    
    async def _process_message(self, topic: str, message: Dict[str, Any]):
        """Process a Kafka message and broadcast via WebSocket"""
        try:
            if topic == "extracted-features":
                # Handle extracted features
                asset_id = message.get('asset_id') or message.get('metadata', {}).get('asset_id')
                if asset_id:
                    await self.websocket_manager.broadcast_feature_update(asset_id, message)
            
            elif topic == "anomalies-detected":
                # Handle anomaly detection
                await self.websocket_manager.broadcast_anomaly_detected(message)
            
            elif topic == "rul-predictions":
                # Handle RUL prediction
                await self.websocket_manager.broadcast_rul_prediction(message)
            
            else:
                logger.debug(f"Topic non géré: {topic}")
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message Kafka: {e}", exc_info=True)
    
    async def stop(self):
        """Stop Kafka consumer"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.consumer:
            self.consumer.close()
        
        logger.info("Consumer Kafka arrêté")

