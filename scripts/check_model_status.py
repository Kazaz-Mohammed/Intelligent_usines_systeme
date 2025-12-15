#!/usr/bin/env python
"""Quick script to check model training status"""
import requests
import json
import sys

def check_status(service_name, url):
    """Check status of a service"""
    try:
        print(f"\n{service_name}:")
        print(f"  URL: {url}")
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  Status: {json.dumps(data, indent=4)}")
        else:
            print(f"  Error: HTTP {r.status_code}")
    except requests.exceptions.Timeout:
        print(f"  Error: Timeout (service may be stuck)")
    except requests.exceptions.ConnectionError:
        print(f"  Error: Connection refused (service may not be running)")
    except Exception as e:
        print(f"  Error: {e}")

# Check both services
check_status("Detection Anomalies", "http://localhost:8084/api/v1/anomalies/status")
check_status("Prediction RUL", "http://localhost:8085/api/v1/rul/status")

