#!/usr/bin/env python
"""Quick test to check if services are accessible"""
import requests

services = {
    "detection-anomalies": "http://localhost:8084/api/v1/anomalies/health",
    "prediction-rul": "http://localhost:8085/api/v1/rul/health",
    "extraction-features": "http://localhost:8083/api/v1/features/features/ENGINE_FD001_000?limit=10"
}

for name, url in services.items():
    try:
        r = requests.get(url, timeout=5)
        if name == "extraction-features":
            data = r.json()
            print(f"{name}: {r.status_code} - Count: {data.get('count', 0)}")
        else:
            print(f"{name}: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"{name}: ERROR - {e}")

