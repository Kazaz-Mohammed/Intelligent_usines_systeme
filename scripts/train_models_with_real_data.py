#!/usr/bin/env python3
"""
Train both RUL and Anomaly Detection models with ALL real extracted features
"""
import requests
import numpy as np
import json
import sys
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration
RUL_API_URL = "http://localhost:8085"
ANOMALY_API_URL = "http://localhost:8084"

# Database connection (using same config as services)
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "predictive_maintenance",
    "user": "pmuser",
    "password": "pmpassword"
}

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)

def fetch_extracted_features():
    """Fetch all extracted features from database"""
    print("=" * 60)
    print("Fetching extracted features from database...")
    print("=" * 60)
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get all features grouped by asset_id, sensor_id, timestamp
        query = """
        SELECT 
            asset_id,
            sensor_id,
            timestamp,
            feature_name,
            feature_value
        FROM extracted_features
        ORDER BY asset_id, sensor_id, timestamp, feature_name
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        print(f"[OK] Fetched {len(rows)} feature records")
        
        # Group by (asset_id, sensor_id, timestamp) to create feature vectors
        feature_vectors = {}
        
        for row in rows:
            key = (row['asset_id'], row['sensor_id'], row['timestamp'])
            if key not in feature_vectors:
                feature_vectors[key] = {
                    'asset_id': row['asset_id'],
                    'sensor_id': row['sensor_id'],
                    'timestamp': row['timestamp'],
                    'features': {}
                }
            feature_vectors[key]['features'][row['feature_name']] = float(row['feature_value'])
        
        print(f"[OK] Created {len(feature_vectors)} feature vectors")
        
        # Convert to lists for training
        feature_names = set()
        for vec in feature_vectors.values():
            feature_names.update(vec['features'].keys())
        
        feature_names = sorted(list(feature_names))
        print(f"[OK] Found {len(feature_names)} unique feature names")
        
        # Create training data arrays
        training_data = []
        metadata = []
        
        for vec in feature_vectors.values():
            # Create feature vector in consistent order
            feature_vector = [vec['features'].get(name, 0.0) for name in feature_names]
            training_data.append(feature_vector)
            metadata.append({
                'asset_id': vec['asset_id'],
                'sensor_id': vec['sensor_id'],
                'timestamp': vec['timestamp'].isoformat() if hasattr(vec['timestamp'], 'isoformat') else str(vec['timestamp'])
            })
        
        return np.array(training_data), feature_names, metadata
        
    finally:
        cursor.close()
        conn.close()

def fetch_rul_data():
    """Fetch RUL predictions to use as training targets"""
    print("\n" + "=" * 60)
    print("Fetching RUL data for training...")
    print("=" * 60)
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get RUL predictions with timestamps
        query = """
        SELECT 
            asset_id,
            sensor_id,
            timestamp,
            rul_mean as rul_prediction
        FROM rul_predictions
        ORDER BY asset_id, sensor_id, timestamp
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        print(f"[OK] Fetched {len(rows)} RUL records")
        
        # Create lookup by (asset_id, sensor_id, timestamp)
        rul_lookup = {}
        for row in rows:
            key = (row['asset_id'], row['sensor_id'], row['timestamp'])
            rul_lookup[key] = float(row['rul_prediction'])
        
        return rul_lookup
        
    finally:
        cursor.close()
        conn.close()

def train_anomaly_models(training_data, feature_names):
    """Train anomaly detection models with real data"""
    print("\n" + "=" * 60)
    print("Training Anomaly Detection Models")
    print("=" * 60)
    
    # Check API health
    try:
        response = requests.get(f"{ANOMALY_API_URL}/health", timeout=5)
        print(f"[OK] Anomaly API: {response.json()['status']}")
    except Exception as e:
        print(f"[ERROR] Anomaly API not accessible: {e}")
        return False
    
    # Prepare training payload
    train_payload = {
        "data": training_data.tolist(),
        "feature_names": feature_names,
        "model_names": ["isolation_forest", "one_class_svm"]
    }
    
    print(f"Training with {len(training_data)} samples, {len(feature_names)} features...")
    
    try:
        response = requests.post(
            f"{ANOMALY_API_URL}/api/v1/anomalies/train",
            json=train_payload,
            timeout=300  # 5 minutes for large dataset
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n[SUCCESS] ANOMALY MODELS TRAINED SUCCESSFULLY")
            for model_name, model_result in result.items():
                if model_result.get('status') == 'success':
                    metrics = model_result.get('metrics', {})
                    print(f"  {model_name}: {metrics.get('n_samples', 'N/A')} samples, "
                          f"anomaly_rate={metrics.get('anomaly_rate', 0):.2%}")
            return True
        else:
            print(f"[ERROR] Training failed: {response.status_code}")
            print(response.text[:500])
            return False
            
    except Exception as e:
        print(f"[ERROR] Training error: {e}")
        return False

def train_rul_models(training_data, feature_names, rul_lookup, metadata):
    """Train RUL prediction models with real data"""
    print("\n" + "=" * 60)
    print("Training RUL Prediction Models")
    print("=" * 60)
    
    # Check API health
    try:
        response = requests.get(f"{RUL_API_URL}/health", timeout=5)
        print(f"[OK] RUL API: {response.json()['status']}")
    except Exception as e:
        print(f"[ERROR] RUL API not accessible: {e}")
        return False
    
    # Match features with RUL values
    target_data = []
    matched_count = 0
    
    for i, meta in enumerate(metadata):
        key = (meta['asset_id'], meta['sensor_id'], meta['timestamp'])
        # Try exact match first
        if key in rul_lookup:
            target_data.append(rul_lookup[key])
            matched_count += 1
        else:
            # Try to find closest timestamp
            asset_id = meta['asset_id']
            sensor_id = meta['sensor_id']
            timestamp = datetime.fromisoformat(meta['timestamp'].replace('Z', '+00:00'))
            
            # Find closest RUL prediction
            closest_rul = None
            min_diff = timedelta.max
            
            for (a_id, s_id, ts), rul_val in rul_lookup.items():
                if a_id == asset_id and s_id == sensor_id:
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    diff = abs((timestamp - ts).total_seconds())
                    if diff < min_diff.total_seconds():
                        min_diff = timedelta(seconds=diff)
                        closest_rul = rul_val
            
            if closest_rul is not None and min_diff.total_seconds() < 3600:  # Within 1 hour
                target_data.append(closest_rul)
                matched_count += 1
            else:
                # Use average RUL if no match
                avg_rul = np.mean(list(rul_lookup.values())) if rul_lookup else 100.0
                target_data.append(avg_rul)
    
    print(f"[OK] Matched {matched_count}/{len(metadata)} feature vectors with RUL values")
    
    # Train LSTM model (needs sequences)
    print("\nTraining LSTM model (sequence-based)...")
    try:
        # Create sequences of length 10
        sequence_length = 10
        sequences = []
        sequence_targets = []
        
        for i in range(len(training_data) - sequence_length + 1):
            seq = training_data[i:i+sequence_length].tolist()
            target = target_data[i+sequence_length-1]
            sequences.append(seq)
            sequence_targets.append(target)
        
        print(f"  Created {len(sequences)} sequences of length {sequence_length}")
        
        lstm_payload = {
            "model_name": "lstm",
            "training_data": sequences,
            "target_data": sequence_targets,
            "feature_names": feature_names,
            "parameters": {
                "sequence_length": sequence_length,
                "epochs": 50,
                "batch_size": 32
            }
        }
        
        response = requests.post(
            f"{RUL_API_URL}/api/v1/rul/train",
            json=lstm_payload,
            timeout=600  # 10 minutes
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  [SUCCESS] LSTM: {result.get('message', 'Trained')}")
        else:
            print(f"  [ERROR] LSTM failed: {response.status_code} - {response.text[:200]}")
    except Exception as e:
        print(f"  [ERROR] LSTM error: {e}")
    
    # Train XGBoost model (single feature vectors)
    print("\nTraining XGBoost model (feature-based)...")
    try:
        xgboost_payload = {
            "model_name": "xgboost",
            "training_data": training_data.tolist(),
            "target_data": target_data,
            "feature_names": feature_names,
            "parameters": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1
            }
        }
        
        response = requests.post(
            f"{RUL_API_URL}/api/v1/rul/train",
            json=xgboost_payload,
            timeout=600  # 10 minutes
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  [SUCCESS] XGBoost: {result.get('message', 'Trained')}")
            return True
        else:
            print(f"  [ERROR] XGBoost failed: {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"  [ERROR] XGBoost error: {e}")
        return False

def main():
    """Main training function"""
    print("=" * 60)
    print("COMPREHENSIVE MODEL TRAINING WITH REAL DATA")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Step 1: Fetch real extracted features
    try:
        training_data, feature_names, metadata = fetch_extracted_features()
        print(f"\n[OK] Training data shape: {training_data.shape}")
        print(f"[OK] Feature names: {len(feature_names)}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch features: {e}")
        sys.exit(1)
    
    # Step 2: Fetch RUL data
    try:
        rul_lookup = fetch_rul_data()
        print(f"[OK] RUL lookup: {len(rul_lookup)} entries")
    except Exception as e:
        print(f"[WARNING] Failed to fetch RUL data: {e}")
        rul_lookup = {}
    
    # Step 3: Train Anomaly Detection models
    anomaly_success = train_anomaly_models(training_data, feature_names)
    
    # Step 4: Train RUL Prediction models
    rul_success = train_rul_models(training_data, feature_names, rul_lookup, metadata)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Anomaly Detection: {'[SUCCESS]' if anomaly_success else '[FAILED]'}")
    print(f"RUL Prediction: {'[SUCCESS]' if rul_success else '[FAILED]'}")
    print(f"\nCompleted at: {datetime.now()}")
    print("\nNEXT STEPS:")
    print("1. Restart the anomaly detection worker:")
    print("   cd services/detection-anomalies")
    print("   python start_worker.py")
    print("2. Restart the RUL prediction worker:")
    print("   cd services/prediction-rul")
    print("   python start_worker.py")
    print("3. Send new data to test the retrained models")
    print("=" * 60)

if __name__ == "__main__":
    main()

