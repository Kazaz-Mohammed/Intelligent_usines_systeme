"""
Test E2E : Flux complet de détection d'anomalie
"""
import pytest
import requests
import time
import json
from confluent_kafka import Producer, Consumer
from typing import Dict, Any

# URLs des services
INGESTION_URL = "http://localhost:8081"
PREPROCESSING_URL = "http://localhost:8082"
EXTRACTION_URL = "http://localhost:8083"
DETECTION_URL = "http://localhost:8084"
ORCHESTRATEUR_URL = "http://localhost:8087"
DASHBOARD_URL = "http://localhost:8086"

# Configuration Kafka
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_RAW_DATA = "raw-sensor-data"
TOPIC_PREPROCESSED = "preprocessed-data"
TOPIC_FEATURES = "extracted-features"
TOPIC_ANOMALIES = "anomalies-detected"
TOPIC_WORK_ORDERS = "work-orders"


class TestE2EAnomalyDetection:
    """Tests E2E pour le flux de détection d'anomalie"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup avant chaque test"""
        # Attendre que les services soient prêts
        self.wait_for_services()
        yield
        # Cleanup après chaque test
        pass
    
    def wait_for_services(self):
        """Attendre que tous les services soient prêts"""
        services = [
            (INGESTION_URL, "IngestionIIoT"),
            (PREPROCESSING_URL, "Preprocessing"),
            (EXTRACTION_URL, "ExtractionFeatures"),
            (DETECTION_URL, "DetectionAnomalies"),
            (ORCHESTRATEUR_URL, "OrchestrateurMaintenance"),
            (DASHBOARD_URL, "DashboardMonitoring")
        ]
        
        for url, name in services:
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    response = requests.get(f"{url}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ {name} est prêt")
                        break
                except:
                    if attempt == max_attempts - 1:
                        pytest.fail(f"{name} n'est pas prêt après {max_attempts} tentatives")
                    time.sleep(2)
    
    def test_complete_anomaly_detection_flow(self):
        """Test du flux complet de détection d'anomalie"""
        
        # 1. Ingestion de données IoT
        print("\n1. Ingestion de données IoT...")
        sensor_data = {
            "assetId": "PUMP_001",
            "sensorId": "SENSOR_001",
            "timestamp": "2024-01-15T10:00:00Z",
            "values": {
                "vibration": 5.2,
                "temperature": 75.5,
                "pressure": 1.2,
                "current": 4.8,
                "rpm": 1500
            }
        }
        
        response = requests.post(
            f"{INGESTION_URL}/api/v1/sensors/data",
            json=sensor_data,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [200, 201], f"Ingestion échouée: {response.text}"
        print("✅ Données ingérées")
        
        # 2. Attendre le prétraitement (via Kafka)
        print("\n2. Attente du prétraitement...")
        time.sleep(5)
        
        # 3. Vérifier l'extraction de caractéristiques
        print("\n3. Vérification de l'extraction de caractéristiques...")
        time.sleep(5)
        
        # 4. Vérifier la détection d'anomalie
        print("\n4. Vérification de la détection d'anomalie...")
        response = requests.get(
            f"{DETECTION_URL}/api/v1/anomalies/",
            params={"asset_id": "PUMP_001", "limit": 10}
        )
        assert response.status_code == 200, f"Détection échouée: {response.text}"
        anomalies = response.json()
        print(f"✅ Anomalies trouvées: {len(anomalies.get('anomalies', []))}")
        
        # 5. Vérifier la création d'intervention
        print("\n5. Vérification de la création d'intervention...")
        time.sleep(5)
        
        response = requests.get(
            f"{ORCHESTRATEUR_URL}/api/v1/work-orders/asset/PUMP_001"
        )
        assert response.status_code == 200, f"Récupération des work orders échouée: {response.text}"
        work_orders = response.json()
        print(f"✅ Work orders trouvés: {len(work_orders)}")
        
        # 6. Vérifier l'affichage dans le dashboard
        print("\n6. Vérification du dashboard...")
        response = requests.get(f"{DASHBOARD_URL}/api/v1/dashboard/overview")
        assert response.status_code == 200, f"Dashboard échoué: {response.text}"
        overview = response.json()
        assert "totalAssets" in overview
        print("✅ Dashboard accessible")
        
        print("\n✅ Flux complet de détection d'anomalie réussi!")
    
    def test_anomaly_detection_via_api(self):
        """Test de détection d'anomalie via API directe"""
        
        # Créer des features directement
        features = {
            "asset_id": "PUMP_002",
            "sensor_id": "SENSOR_002",
            "timestamp": "2024-01-15T10:00:00Z",
            "features": {
                "rms": 10.5,
                "kurtosis": 2.3,
                "skewness": 0.8,
                "peak": 15.2,
                "crest_factor": 3.5
            }
        }
        
        # Détecter l'anomalie
        response = requests.post(
            f"{DETECTION_URL}/api/v1/anomalies/detect",
            json=features
        )
        
        assert response.status_code == 200, f"Détection échouée: {response.text}"
        result = response.json()
        assert "is_anomaly" in result
        assert "final_score" in result
        print(f"✅ Anomalie détectée: {result.get('is_anomaly')}, Score: {result.get('final_score')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

