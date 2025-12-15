#!/usr/bin/env python
"""Simple script to start the dashboard-usine service"""
import sys
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("Starting dashboard-usine service on port 8091...")
    print("Press Ctrl+C to stop")
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=settings.service_port,
            reload=False,
            log_level=settings.log_level.lower()
        )
    except KeyboardInterrupt:
        print("\nService stopped")
    except Exception as e:
        print(f"Error starting service: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

