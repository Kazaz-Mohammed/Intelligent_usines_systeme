#!/usr/bin/env python3
"""
Retrain anomaly detection models with correct feature dimensions
"""
import requests
import numpy as np
import json

# Configuration
ANOMALY_API_URL = "http://localhost:8084"
NUM_FEATURES = 90  # Match the extracted features count

def generate_training_data(num_samples=500, num_features=90):
    """Generate realistic training data with correct feature dimensions"""
    print(f"Generating {num_samples} samples with {num_features} features...")
    
    # Create feature names matching extracted features pattern
    feature_names = []
    sensors = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 
               'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 
               'PCNfR_dmd', 'W31', 'W32', 'OP_SETTING_1', 'OP_SETTING_2', 'OP_SETTING_3']
    
    feature_types = ['mean', 'std', 'min', 'max']
    
    for sensor in sensors:
        for ftype in feature_types:
            feature_names.append(f"{sensor}_{ftype}")
            if len(feature_names) >= num_features:
                break
        if len(feature_names) >= num_features:
            break
    
    # Pad with generic names if needed
    while len(feature_names) < num_features:
        feature_names.append(f"feature_{len(feature_names)}")
    
    feature_names = feature_names[:num_features]
    
    # Generate normal operation data (majority)
    np.random.seed(42)
    normal_samples = int(num_samples * 0.95)
    anomaly_samples = num_samples - normal_samples
    
    # Normal data - sensor readings in typical ranges
    normal_data = np.random.randn(normal_samples, num_features) * 0.5 + 0.5
    
    # Anomaly data - values outside normal range
    anomaly_data = np.random.randn(anomaly_samples, num_features) * 2 + 3
    
    # Combine
    training_data = np.vstack([normal_data, anomaly_data])
    
    # Shuffle
    indices = np.random.permutation(num_samples)
    training_data = training_data[indices]
    
    return training_data.tolist(), feature_names

def retrain_models():
    """Retrain all anomaly detection models"""
    
    # Check API health
    try:
        response = requests.get(f"{ANOMALY_API_URL}/health", timeout=5)
        print(f"Anomaly API health: {response.json()}")
    except Exception as e:
        print(f"ERROR: Anomaly API not accessible at {ANOMALY_API_URL}: {e}")
        return False
    
    # Generate training data
    training_data, feature_names = generate_training_data(500, NUM_FEATURES)
    
    print(f"\nTraining with {len(training_data)} samples, {len(feature_names)} features")
    
    # Train models via API - use correct format
    train_payload = {
        "data": training_data,
        "feature_names": feature_names,
        "model_names": ["isolation_forest", "one_class_svm"]
    }
    
    print("\nTraining anomaly detection models...")
    try:
        response = requests.post(
            f"{ANOMALY_API_URL}/api/v1/anomalies/train",
            json=train_payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== TRAINING SUCCESSFUL ===")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Training failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Training error: {e}")
        return False

def test_prediction():
    """Test anomaly prediction with new data"""
    print("\n=== Testing Anomaly Prediction ===")
    
    # Generate test sample (normal)
    test_normal = np.random.randn(1, NUM_FEATURES) * 0.5 + 0.5
    
    # Generate test sample (anomaly)
    test_anomaly = np.random.randn(1, NUM_FEATURES) * 3 + 5
    
    predict_url = f"{ANOMALY_API_URL}/api/v1/anomalies/detect"
    
    # Test normal sample
    print("\nTesting normal sample...")
    try:
        response = requests.post(
            predict_url,
            json={
                "features": test_normal.tolist()[0],
                "asset_id": "TEST_ASSET",
                "sensor_id": "TEST_SENSOR"
            },
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  Normal sample result: is_anomaly={result.get('is_anomaly')}, score={result.get('anomaly_score', 'N/A')}")
        else:
            print(f"  Prediction failed: {response.status_code} - {response.text[:200]}")
    except Exception as e:
        print(f"  Prediction error: {e}")
    
    # Test anomaly sample
    print("\nTesting anomaly sample...")
    try:
        response = requests.post(
            predict_url,
            json={
                "features": test_anomaly.tolist()[0],
                "asset_id": "TEST_ASSET",
                "sensor_id": "TEST_SENSOR"
            },
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"  Anomaly sample result: is_anomaly={result.get('is_anomaly')}, score={result.get('anomaly_score', 'N/A')}")
        else:
            print(f"  Prediction failed: {response.status_code} - {response.text[:200]}")
    except Exception as e:
        print(f"  Prediction error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Retraining Anomaly Detection Models")
    print("=" * 60)
    
    if retrain_models():
        test_prediction()
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("1. Restart the anomaly detection worker:")
        print("   cd services/detection-anomalies")
        print("   python start_worker.py")
        print("2. Send new data to test")
        print("=" * 60)
    else:
        print("\nTraining failed. Check the anomaly API logs.")

