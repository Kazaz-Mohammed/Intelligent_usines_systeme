#!/usr/bin/env python3
"""Check what's in the saved anomaly detection models"""
import joblib
from pathlib import Path

models_dir = Path("services/detection-anomalies/models")

print("=" * 60)
print("Checking saved anomaly detection models")
print("=" * 60)

for model_file in models_dir.glob("*.pkl"):
    print(f"\n{model_file.name}:")
    try:
        data = joblib.load(model_file)
        feature_names = data.get("feature_names", [])
        is_trained = data.get("is_trained", False)
        print(f"  - Is trained: {is_trained}")
        print(f"  - Number of features: {len(feature_names)}")
        if feature_names:
            print(f"  - First 5 features: {feature_names[:5]}")
            print(f"  - Last 5 features: {feature_names[-5:]}")
    except Exception as e:
        print(f"  - Error loading: {e}")

print("\n" + "=" * 60)

