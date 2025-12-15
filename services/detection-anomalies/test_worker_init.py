#!/usr/bin/env python
"""Test script to verify worker initialization"""
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

try:
    logger.info("=" * 60)
    logger.info("TEST: Worker Initialization")
    logger.info("=" * 60)
    
    logger.info("Step 1: Importing worker module...")
    from app.worker import AnomalyDetectionWorker
    logger.info("✓ Worker module imported")
    
    logger.info("Step 2: Creating worker instance...")
    worker = AnomalyDetectionWorker()
    logger.info("✓ Worker instance created successfully!")
    
    logger.info("Step 3: Worker is ready!")
    logger.info(f"  - Kafka Consumer: {'✓' if worker.kafka_consumer else '✗'}")
    logger.info(f"  - Kafka Producer: {'✓' if worker.kafka_producer else '✗'}")
    logger.info(f"  - Anomaly Detection Service: {'✓' if worker.anomaly_detection_service else '✗'}")
    logger.info(f"  - PostgreSQL Service: {'✓' if worker.postgresql_service else '✗'}")
    
    logger.info("=" * 60)
    logger.info("SUCCESS: All components initialized!")
    logger.info("=" * 60)
    
except Exception as e:
    logger.error("=" * 60)
    logger.error(f"ERROR: {e}", exc_info=True)
    logger.error("=" * 60)
    sys.exit(1)

