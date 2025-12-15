#!/usr/bin/env python
"""
Standalone script to run the detection-anomalies worker
"""
import sys
import logging
from app.worker import AnomalyDetectionWorker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    print("=" * 60)
    print("Starting Anomaly Detection Worker")
    print("=" * 60)
    logger.info("=" * 60)
    logger.info("Starting Anomaly Detection Worker")
    logger.info("=" * 60)
    
    try:
        print("Creating worker instance...")
        logger.info("Creating worker instance...")
        worker = AnomalyDetectionWorker()
        print("Worker instance created successfully!")
        logger.info("Worker instance created successfully")
        
        print("Starting worker...")
        logger.info("Starting worker...")
        worker.start()
    except KeyboardInterrupt:
        print("\nInterruption received")
        logger.info("Interruption received")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
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

