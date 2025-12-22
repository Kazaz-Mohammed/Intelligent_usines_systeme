#!/usr/bin/env python
"""
Standalone script to run the extraction-features worker
"""
import sys
import os

# Ensure the current directory is in sys.path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import logging

# Debug: verify ExtractedFeature model
from app.models.feature_data import ExtractedFeature
print(f"ExtractedFeature fields: {list(ExtractedFeature.model_fields.keys())}")
print(f"ExtractedFeature module: {ExtractedFeature.__module__}")

from app.worker import FeatureExtractionWorker
from app.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    import sys
    
    print("=" * 60)
    print("Starting Extraction Features Worker")
    print("=" * 60)
    logger.info("=" * 60)
    logger.info("Starting Extraction Features Worker")
    logger.info("=" * 60)
    logger.info(f"Consuming from: {settings.kafka_topic_input_preprocessed}")
    logger.info(f"Publishing to: {settings.kafka_topic_output}")
    logger.info("=" * 60)
    
    try:
        print("Creating worker instance...")
        worker = FeatureExtractionWorker()
        print("Worker instance created successfully!")
        
        print("Starting worker...")
        logger.info("Starting worker...")
        worker.start(mode="streaming")
        print("Worker started. Waiting for messages...")
        print("Press Ctrl+C to stop")
        logger.info("Worker started. Waiting for messages...")
        
        # Keep the main thread alive
        import time
        while worker.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterruption received")
        logger.info("Interruption received")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("Stopping worker...")
        logger.info("Stopping worker...")
        try:
            worker.stop()
        except:
            pass
        print("Worker stopped")
        logger.info("Worker stopped")

if __name__ == "__main__":
    main()

