#!/usr/bin/env python3
"""
Train anomaly detection models using real extracted features from the database.

This script:
1. Fetches real extracted features from PostgreSQL
2. Prepares the data for anomaly detection (unsupervised learning)
3. Trains Isolation Forest and One-Class SVM models via the API
4. The models will learn "normal" patterns from the data
"""
import requests
import numpy as np
import psycopg2
import json
from collections import defaultdict

# Configuration
ANOMALY_API_URL = "http://localhost:8084"
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "predictive_maintenance",
    "user": "pmuser",
    "password": "pmpassword"
}

def fetch_features_from_db():
    """Fetch extracted features from the database"""
    print("Connecting to PostgreSQL...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Get feature statistics
    cursor.execute("SELECT COUNT(*) FROM extracted_features")
    total_count = cursor.fetchone()[0]
    print(f"Total features in database: {total_count}")
    
    # Get unique feature names
    cursor.execute("""
        SELECT DISTINCT feature_name 
        FROM extracted_features 
        ORDER BY feature_name
    """)
    all_feature_names = [row[0] for row in cursor.fetchall()]
    print(f"Unique feature names: {len(all_feature_names)}")
    
    # Get features grouped by asset_id, sensor_id, and timestamp window
    # We'll create samples by combining features from the same time window
    print("Fetching features (this may take a moment)...")
    
    cursor.execute("""
        SELECT 
            asset_id,
            sensor_id,
            DATE_TRUNC('minute', timestamp) as time_bucket,
            feature_name,
            feature_value
        FROM extracted_features
        WHERE feature_value IS NOT NULL 
          AND feature_value != 'NaN'::float
          AND ABS(feature_value) < 1e10
        ORDER BY asset_id, sensor_id, time_bucket, feature_name
        LIMIT 500000
    """)
    
    rows = cursor.fetchall()
    print(f"Fetched {len(rows)} feature records")
    
    cursor.close()
    conn.close()
    
    return rows, all_feature_names

def prepare_training_data(rows, target_features=90):
    """Prepare training data from raw feature rows"""
    print("\nPreparing training data...")
    
    # Group features by (asset_id, sensor_id, time_bucket)
    samples = defaultdict(dict)
    
    for asset_id, sensor_id, time_bucket, feature_name, feature_value in rows:
        key = (asset_id, sensor_id, str(time_bucket))
        samples[key][feature_name] = float(feature_value)
    
    print(f"Created {len(samples)} sample groups")
    
    # Find the most common feature names across all samples
    feature_counts = defaultdict(int)
    for sample in samples.values():
        for fname in sample.keys():
            feature_counts[fname] += 1
    
    # Sort features by frequency and take top N
    sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])
    top_features = [f[0] for f in sorted_features[:target_features]]
    
    print(f"Selected top {len(top_features)} features by frequency")
    print(f"Top 10 features: {top_features[:10]}")
    
    # Convert to numpy array
    X = []
    valid_samples = 0
    skipped_samples = 0
    
    for key, features in samples.items():
        # Only include samples that have most of the top features
        available = sum(1 for f in top_features if f in features)
        if available >= len(top_features) * 0.7:  # At least 70% of features present
            row = []
            for fname in top_features:
                val = features.get(fname, 0.0)
                # Handle NaN and infinity
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                row.append(val)
            X.append(row)
            valid_samples += 1
        else:
            skipped_samples += 1
    
    print(f"Valid samples: {valid_samples}, Skipped: {skipped_samples}")
    
    X = np.array(X)
    
    # Normalize the data (important for anomaly detection)
    print("Normalizing data...")
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds[stds == 0] = 1  # Avoid division by zero
    X_normalized = (X - means) / stds
    
    # Replace any remaining NaN/inf
    X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Final training data shape: {X_normalized.shape}")
    print(f"Data stats - Mean: {np.mean(X_normalized):.4f}, Std: {np.std(X_normalized):.4f}")
    print(f"Data range: [{np.min(X_normalized):.4f}, {np.max(X_normalized):.4f}]")
    
    return X_normalized, top_features

def train_models(X, feature_names):
    """Train anomaly detection models via API"""
    print("\n" + "=" * 60)
    print("Training Anomaly Detection Models")
    print("=" * 60)
    
    # Check API health
    try:
        response = requests.get(f"{ANOMALY_API_URL}/health", timeout=5)
        print(f"Anomaly API health: {response.json()}")
    except Exception as e:
        print(f"[ERROR] Anomaly API not accessible: {e}")
        return False
    
    # Prepare training payload
    # For anomaly detection, we train on "normal" data
    # The models will learn the normal patterns and flag deviations
    
    # Use a subset if too large
    max_samples = 5000
    if len(X) > max_samples:
        print(f"Subsampling from {len(X)} to {max_samples} samples...")
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_train = X[indices]
    else:
        X_train = X
    
    print(f"\nTraining with {len(X_train)} samples, {len(feature_names)} features")
    
    train_payload = {
        "data": X_train.tolist(),
        "feature_names": feature_names,
        "model_names": ["isolation_forest", "one_class_svm"]
    }
    
    print("\nSending training request to API...")
    try:
        response = requests.post(
            f"{ANOMALY_API_URL}/api/v1/anomalies/train",
            json=train_payload,
            timeout=300  # 5 minutes timeout for training
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== TRAINING SUCCESSFUL ===")
            print(json.dumps(result, indent=2, default=str))
            return True
        else:
            print(f"[ERROR] Training failed: {response.status_code}")
            print(response.text[:500])
            return False
            
    except Exception as e:
        print(f"[ERROR] Training error: {e}")
        return False

def test_detection():
    """Test anomaly detection with sample data"""
    print("\n" + "=" * 60)
    print("Testing Anomaly Detection")
    print("=" * 60)
    
    # Check model status
    try:
        response = requests.get(f"{ANOMALY_API_URL}/api/v1/anomalies/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"Model status: {json.dumps(status, indent=2)}")
        else:
            print(f"Status check failed: {response.status_code}")
    except Exception as e:
        print(f"Status check error: {e}")

def main():
    print("=" * 60)
    print("Training Anomaly Models with Real Extracted Features")
    print("=" * 60)
    
    # Step 1: Fetch features from database
    rows, all_feature_names = fetch_features_from_db()
    
    if len(rows) == 0:
        print("[ERROR] No features found in database!")
        return
    
    # Step 2: Prepare training data
    X, feature_names = prepare_training_data(rows, target_features=90)
    
    if len(X) < 100:
        print(f"[ERROR] Not enough samples ({len(X)}). Need at least 100.")
        return
    
    # Step 3: Train models
    if train_models(X, feature_names):
        print("\n[SUCCESS] Models trained successfully!")
        
        # Step 4: Test detection
        test_detection()
        
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("1. RESTART the anomaly detection worker:")
        print("   - Stop the current worker (Ctrl+C in terminal 12)")
        print("   - cd services/detection-anomalies")
        print("   - python start_worker.py")
        print("")
        print("2. Send new data to see anomaly detection in action:")
        print("   cd datasets/nasa-cmapss")
        print("   python load_and_send_cmapss.py --file test_FD001.txt --max-units 3 --max-cycles 20")
        print("=" * 60)
    else:
        print("\n[ERROR] Training failed!")

if __name__ == "__main__":
    main()

