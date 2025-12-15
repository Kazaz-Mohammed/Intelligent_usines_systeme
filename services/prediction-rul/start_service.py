#!/usr/bin/env python
"""Simple script to start the prediction-rul service"""
import sys
import uvicorn

if __name__ == "__main__":
    print("Starting prediction-rul service on port 8085...")
    print("Press Ctrl+C to stop")
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8085,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nService stopped")
    except Exception as e:
        print(f"Error starting service: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

