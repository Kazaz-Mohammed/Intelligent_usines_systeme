#!/usr/bin/env python3
"""Test anomaly detection directly via API"""
import requests
import json
import psycopg2

API_URL = "http://localhost:8084"
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "predictive_maintenance",
    "user": "pmuser",
    "password": "pmpassword"
}

def get_sample_features():
    """Get sample extracted features from database"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Get a sample of recent features grouped by asset/sensor/time
    cursor.execute("""
        SELECT 
            asset_id,
            sensor_id,
            DATE_TRUNC('minute', timestamp) as time_bucket,
            json_object_agg(feature_name, feature_value) as features
        FROM extracted_features
        WHERE feature_value IS NOT NULL 
          AND ABS(feature_value) < 1e15
        GROUP BY asset_id, sensor_id, DATE_TRUNC('minute', timestamp)
        ORDER BY time_bucket DESC
        LIMIT 5
    """)
    
    samples = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return samples

def test_detection(asset_id, sensor_id, features):
    """Test detection via API"""
    payload = {
        "asset_id": asset_id,
        "sensor_id": sensor_id,
        "features": features,
        "timestamp": "2025-12-20T12:00:00Z"
    }
    
    try:
        response = requests.post(
            f"{API_URL}/api/v1/anomalies/detect",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"  [ERROR] Status {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

def main():
    print("=" * 60)
    print("Testing Anomaly Detection API")
    print("=" * 60)
    
    # Check API status first
    try:
        status = requests.get(f"{API_URL}/api/v1/anomalies/status", timeout=5).json()
        print(f"\nAPI Status: ready={status.get('ready')}")
        for model_name, model_info in status.get('models', {}).items():
            print(f"  - {model_name}: trained={model_info.get('is_trained')}, features={model_info.get('n_features')}")
    except Exception as e:
        print(f"[ERROR] Cannot reach API: {e}")
        return
    
    print("\n" + "-" * 60)
    print("Fetching sample features from database...")
    samples = get_sample_features()
    print(f"Found {len(samples)} sample feature groups")
    
    print("\n" + "-" * 60)
    print("Testing detection on samples:")
    
    for i, (asset_id, sensor_id, time_bucket, features_json) in enumerate(samples):
        features = features_json if isinstance(features_json, dict) else json.loads(features_json)
        print(f"\n[Sample {i+1}] {asset_id}/{sensor_id} ({len(features)} features)")
        
        result = test_detection(asset_id, sensor_id, features)
        
        if result:
            is_anomaly = result.get('is_anomaly')
            final_score = result.get('final_score', 0)
            criticality = result.get('criticality')
            scores = result.get('scores', [])
            
            print(f"  -> is_anomaly: {is_anomaly}")
            print(f"  -> final_score: {final_score:.4f}")
            print(f"  -> criticality: {criticality}")
            print(f"  -> model scores:")
            for s in scores:
                print(f"       - {s['model_name']}: score={s['score']:.4f}, threshold={s['threshold']:.4f}, is_anomaly={s['is_anomaly']}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

